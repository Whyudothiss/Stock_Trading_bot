import os
import logging
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from pymongo import MongoClient
from bson import ObjectId
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Logging set Up
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
BOT_Token = os.getenv("BOT_TOKEN")

if not MONGO_URI or not BOT_Token:
    raise ValueError("Please set MONGO_URI and BOT_TOKEN environment variables")

# MongoDB setup
try:
    from urllib.parse import quote_plus
    # If URI contains username and password, encode them
    if '@' in MONGO_URI:
        # Split the URI into parts
        prefix = MONGO_URI.split('://', 1)[0]
        rest = MONGO_URI.split('://', 1)[1]
        credentials = rest.split('@')[0]
        host_part = rest.split('@')[1]
        username = credentials.split(':')[0]
        password = credentials.split(':')[1]
        
        # URL encode the username and password
        encoded_username = quote_plus(username)
        encoded_password = quote_plus(password)
        
        # Reconstruct the URI
        MONGO_URI = f"{prefix}://{encoded_username}:{encoded_password}@{host_part}"

    # Connect with SSL settings for Atlas
    client = MongoClient(
        MONGO_URI,
        tlsAllowInvalidCertificates=True  # This bypasses certificate verification
    )
    # Test the connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    raise
db = client.portfoloio_tracker

# Collections
users_collection = db.users
transactions_collection = db.transactions

(TICKER, QUANTITY, PRICE, TRIM_TYPE, TRIM_VALUE, 
 DELETE_CONFIRM, EDIT_SELECT, EDIT_FIELD, EDIT_VALUE) = range(9)

class PortfolioTracker():
    def __init__(self):
        self.user_data = {}
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        username = update.effective_user.username or update.effective_user.first_name

        # Ensure user is in the data dictionary
        if not users_collection.find_one({"user_id": user_id}):
            users_collection.insert_one({
                "user_id": user_id,
                "username": username,
                "create_at": datetime.now()
            })

        welcome_msg = """ 🎯 Welcome to myPortfolioTracker! 🎯

Your personal stock portfolio management bot. Here are the available commands:

📊 Portfolio Commands:
/buy - Record a stock purchase
/sell - Record a stock sale  
/trim - Partial sell (by % or quantity)
/current - View current holdings
/past - View sold positions with P&L
/pnl - Overall profit/loss summary
/portfolio - Quick portfolio overview

🛠 Management Commands:
/delete - Delete a transaction
/edit - Edit a transaction
/help - Show this help message

Ready to track your investments! Use /buy to record your first purchase."""


        await update.message.reply_text(welcome_msg)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.start(update, context)


    # Buy transaction flow
    async def buy_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start buy trasnsaction"""
        context.user_data['action'] = 'buy'
        await update.message.reply_text("Let's record a stock purchase!\n\n Whats the ticker symbol (e.g., AAPL):")
        return TICKER
    
    async def get_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Get ticker symbol"""
        ticker = update.message.text.strip().upper()
        context.user_data['ticker'] = ticker

        await update.message.reply_text(f"Stock: {ticker}\n\n How many shares did you buy?")
        return QUANTITY
    
    async def get_quantity(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            quantity = float(update.message.text)

            if quantity <= 0:
                await update.message.reply_text("Please enter a positive number")
                return QUANTITY

            context.user_data['quantity'] = quantity
            ticker = context.user_data['ticker']
            action = context.user_data['action']

            await update.message.reply_text(f"Stock: {ticker}\nQuantity: {quantity} shares\n\n What price did you {'buy' if action == 'buy' else 'sell'} at? (USD)")
            return PRICE
        
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return QUANTITY
            
    async def get_price_and_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            price = float(update.message.text)    
            if price <= 0:
                await update.message.reply_text("PLease enter a positive price")
                return PRICE

            user_id = update.effective_user.id
            ticker = context.user_data['ticker']
            quantity = context.user_data['quantity']
            action = context.user_data['action']
            total_value = quantity * price

            # Check for sell/trim whether user has enough shares
            if action in ['sell', 'trim']:
                current_position = self.get_current_positions(user_id, ticker)
                if current_position < quantity:
                    await update.message.reply_test(f"❌ Error: You only own {current_position} shares of {ticker}, "
                        f"but trying to {action} {quantity} shares.\n\nTransaction cancelled.")
                    return ConversationHandler.END

            transaction ={
                "user_id": user_id,
                "ticker": ticker,
                "action": action,
                "price": price,
                "quantity": quantity,
                "total_value": total_value,
                "timestamp": datetime.now()
            }

            transactions_collection.insert_one(transaction)

            action_emoji = "📈" if action == "buy" else "📉"
            confirmation = f"""
{action_emoji} Transaction Recorded!
Stock: {ticker}
Action: {action.upper()}
Quantity: {quantity} shares
Price: ${price:.2f}
Total: ${total_value:.2f}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

✅ Successfully added to your portfolio! 
           """
            await update.message.reply_text(confirmation)
            return ConversationHandler.END
        except ValueError:
            await update.message.reoly_text("Please enter a valid price)")
            return PRICE

    async def sell_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start sell transaction"""
        context.user_data['action'] = 'sell'
        await update.message.reply_text("📉 Let's record a stock sale!\n\nWhat's the ticker symbol?")
        return TICKER

    async def trim_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start trim transaction"""
        await update.message.reply_text("✂️ Let's record a stock trim (partial sell)!\n\nWhat's the ticker symbol?")
        return TICKER

    async def get_trim_ticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Get ticker symbol for trim"""
        ticker = update.message.text.strip().upper()
        user_id = update.effective_user.id

        current_position = self.get_current_positions(user_id, ticker)
        if current_position <= 0:
            await update.message.reply_text(f"❌ You don't currently own any {ticker} shares.")
            return ConversationHandler.END
        
        context.user_data['ticker'] = ticker
        context.user_data['current_position'] = current_position

        keyboard = [['Percentage', 'Quantity']]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)

        await update.message.reply_test(
            f"Stock: {ticker}\nCurrent position: {current_position} shares\n\n"
            "How would you like to trim?",
            reply_markup=reply_markup
        )
        return TRIM_TYPE
    
    async def get_trim_type(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        trim_type = update.message.text.strip().lower()
        if trim_type not in ['percentage', 'quantity']:
            await update.message.reply_text("Please choose either Percentage or Quantity")
            return TRIM_TYPE
        ticker = context.user_data['ticker']
        context.user_data['trim_type'] = trim_type
        current_position = context.user_data['current)position']

        if trim_type == 'percentage':
            await update.message.reply_test(
                f"Stock: {ticker}\nCurrent: {current_position} shares\n\n"
                "What percentage would you like to sell? (e.g., 25 for 25%)",
                reply_markup=ReplyKeyboardRemove()
            )
        else:
            await update.message.reply_text(
                f"Stock: {ticker}\nCurrent: {current_position} shares\n\n"
                "How many shares would you like to sell?",
                reply_markup=ReplyKeyboardRemove()
            )
        return TRIM_VALUE

    async def get_trim_value(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        try:
            value = float(update.message.text)
            trim_type = context.user_data['trim_type']
            ticker = context.user_data['ticker']
            current_position = context.user_data['current_position']

            # For percentage trim type
            if trim_type == 'percentage':
                if value <= 0 or value > 100:
                    await update.message.reply_text("Please enter a percentage between 0 and 100")
                    return TRIM_VALUE
                quantity = (value / 100) * current_position
            # For quantity trim type
            else:
                if value <= 0 value > current_position:
                    await update.message.reply_text(f"Please enter a quantity between 0 and {current_position}")
                    return TRIM_VALUE
                quantity = value
            
            context.user_data['quantity'] = quantity
            context.user_data['action'] = 'trim'

            await update.message.reply_text(f"Stock: {ticker}\n"
                f"Selling: {quantity:.2f} shares ({(quantity/current_position)*100:.1f}%)\n"
                f"Remaining: {current_position - quantity:.2f} shares\n\n"
                "At what price are you selling? (USD)"
            )
            return PRICE
        except ValueError:
            await update.message.reply_text("Please enter a valid number")
            return TRIM_VALUE

    def get_current_positions(self, user_id, ticker):
        """Get current positions for a user and ticker"""
        pipeline = [
            {"$match": {"user_id": user_id, "ticker": ticker}},
            {"$group": {
                "_id": "$ticker",
                "total_bought": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$quantity", 0]}}
            }}
        ]

        result = list(transactions_collection.aggregate(pipeline))
        if result:
            return result[0]["total_bought"] - result[0]["total_sold"]
        return 0
    
    def get_user_holdings(self, user_id):
        """Get all current holdings for a user"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$ticker",
                "total_bought": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$quantity", 0]}},
                "total_sold": {"$sum": {"$cond": [{"$neq": ["$action", "buy"]}, "$quantity", 0]}}
            }}
        ]

        results = list(transactions_collection.aggregate(pipeline))
        holdings = {}
        for result in results:
            current_qty = result["total_bought"] - result["total_sold"]
            if current_qty > 0:
                holdings[result["_id"]] = current_qty
        
        return holdings
    
    async def current_holdings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current holdings"""
        user_id = update.effective_user.id
        holdings = self.get_user_holdings(user_id)

        if not holdings:
            await update.message.reply_text("📊 You don't have any current holdings.")
            return
        
        message = "📊 Your Current Holdings:\n\n"
        for ticker, data in holdings.items():
            message += f"🏷 {ticker}: {data['quantity']} shares\n"
        
        await update.message.reply_text(message)
       
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel the conversation"""
        await update.message.reply_text(
            "❌ Operation cancelled.",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

def main():
    tracker = PortfolioTracker()

    # Create application
    application = Application.builder().token(BOT_Token).build()

    # Buy conversation handler
    buy_handler = ConversationHandler(
        entry_points = [CommandHandler('buy', tracker.buy_start)],
        states = {
            TICKER: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_ticker)],
            QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_quantity)],
            PRICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, tracker.get_price_and_execute)],     
        },
        fallbacks = [CommandHandler('cancel', tracker.cancel)],
    )

    # Add handlers
    application.add_handler(CommandHandler("start", tracker.start))
    application.add_handler(CommandHandler("help", tracker.help_command))
    application.add_handler(buy_handler)
    application.add_handler(CommandHandler("current", tracker.current_holdings))

    print("Bot started...")
    application.run_polling()
    
if __name__ == '__main__':
    main()    