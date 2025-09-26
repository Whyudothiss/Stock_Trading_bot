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
            await update.message.reply_text(f"✅ {ticker} purchase recorded!")
            return ConversationHandler.END
        except ValueError:
            await update.message.reoly_text("Please enter a valid price)")
            return PRICE

    async def sell_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start sell transaction"""
        context.user_data['action'] = 'sell'
        await update.message.reply_text("📉 Let's record a stock sale!\n\nWhat's the ticker symbol?")
        return TICKER
    
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
                "total_sold": {"$sum": {"$cond": [{"$ne": ["$action", "buy"]}, "$quantity", 0]}},
                "avg_buy_price": {"$avg": {"$cond": [{"$eq": ["$action", "buy"]}, "$price", 0]}},
                "total_value": {"$sum": {"$cond": [{"$eq": ["$action", "buy"]}, "$total_value", 0]}}
            }},
            {"$match": {
                "$expr": {"$gt": [{"$subtract": ["$total_bought", "$total_sold"]}, 0]}
            }}
        ]
        return list(transactions_collection.aggregate(pipeline))

    async def current_holdings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current holdings command"""
        user_id = update.effective_user.id
        holdings = self.get_user_holdings(user_id)

        if not holdings:
            await update.message.reply_text("You don't have any current holdings.")
            return

        message = "📊 Current Holdings:\n\n"
        total_portfolio_value = 0

        for holding in holdings:
            ticker = holding["_id"]
            shares = holding["total_bought"] - holding["total_sold"]
            avg_price = holding["avg_buy_price"]
            position_value = shares * avg_price
            total_portfolio_value += position_value

            message += f"🔸 {ticker}:\n"
            message += f"   Shares: {shares:.2f}\n"
            message += f"   Avg Price: ${avg_price:.2f}\n"
            message += f"   Position Value: ${position_value:.2f}\n\n"

        message += f"Total Portfolio Value: ${total_portfolio_value:.2f}"
        
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